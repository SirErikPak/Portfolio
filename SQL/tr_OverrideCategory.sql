IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_OverrideCategory]'))
DROP TRIGGER [dbo].[tr_OverrideCategory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_OverrideCategory] ON [dbo].[OverrideCategory] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info			Varchar(25),
			@SeqNo			Int,
			@Max			Int,
			@Min			Int,
			@iEmpId			Varchar(10),
			@iCategoryCode	Varchar(10),
			@iCategoryValue	Varchar(25),
			@iLastUser		Int,
			@iLastUpdate	Char(10),
			@cEmpId			Varchar(10),
			@cCategoryCode	Varchar(10),
			@cCategoryValue	Varchar(25),
			@cLastUser		Int,
			@cLastUpdate	Char(10)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(OC.EmpId) 
	From OverrideCategory OC INNER JOIN deleted DEL 
	ON OC.EmpId = DEL.EmpId 
	AND	OC.CategoryCode = DEL.CategoryCode) > 0

	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo			Int		Identity,
			EmpID			Varchar(10)	NOT NULL,
			CategoryCode	Varchar(10) NOT NULL
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	EmpID,
				CategoryCode
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(EmpID),
				@SeqNo	= 1,
				@Min	= 1
		From	#Temp

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		WHILE @SeqNo <= @Max
			 Begin		-- BEGIN WHILE LOOP		
		
			/*********************************************************/
			/*                                                       */
			/*********************************************************/		
			SELECT	@cEmpId			= O.EmpId,
					@cCategoryCode	= O.CategoryCode,
					@cCategoryValue	= O.CategoryValue,
					@cLastUser		= O.LastUser,
					@cLastUpdate	= CONVERT(Char(10),O.LastUpdate,101)
			From	OverrideCategory O INNER JOIN deleted D 
					ON O.EmpId = D.EmpId 
					AND	O.CategoryCode = D.CategoryCode
			Where	O.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
			And		O.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
								
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			SELECT	@iEmpId			= D.EmpId,
					@iCategoryCode	= D.CategoryCode,
					@iCategoryValue	= D.CategoryValue,
					@iLastUser		= D.LastUser,
					@iLastUpdate	= CONVERT(Char(10),D.LastUpdate,101)
			From	OverrideCategory O INNER JOIN deleted D 
					ON O.EmpId = D.EmpId 
					AND	O.CategoryCode = D.CategoryCode	
			Where	D.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
			And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
				
		/*************************************************************/
		/*                                                           */
		/*************************************************************/	
		IF	(
			-----(@cEmpId = @iEmpId) And (@cCategoryCode	= @iCategoryCode) And 
			(@cCategoryValue = @iCategoryValue) And
			(@cLastUser = @iLastUser)----- And (@cLastUpdate	= @iLastUpdate)				
			)
			Begin
				/*****************************************************/
				/*					No Change(s)                     */
				/*****************************************************/
				GOTO NextRecordCheck
					
			End
		Else
			Begin
				/*****************************************************/
				/*				Update Change(s)                     */
				/*****************************************************/
				Insert	AuditOverrideCategory
				SELECT	D.EmpId,
						D.CategoryCode,
						D.CategoryValue,
						D.LastUser,
						D.LastUpdate,
						@Info,
						GETDATE(),
						USER_NAME()
				From	OverrideCategory O INNER JOIN deleted D 
						ON O.EmpId = D.EmpId 
						AND	O.CategoryCode = D.CategoryCode
				Where	D.EmpId IN (Select EmpId From #Temp Where SeqNo = @SeqNo)
				And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
								
			End

				/*****************************************************/
				/*                                                   */
				/*****************************************************/
				NextRecordCheck:	
				Select	@SeqNo = @SeqNo + @Min

			 End		-- END WHILE LOOP
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Drop Table #Temp
							
	End
 Else
	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Delete'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	AuditOverrideCategory
		SELECT	EmpId,
				CategoryCode,
				CategoryValue,
				LastUser,
				LastUpdate,
				@Info,
				GETDATE(),
				USER_NAME()
		From	Deleted
		
	End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO


