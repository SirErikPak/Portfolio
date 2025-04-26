IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_AppUser]'))
DROP TRIGGER [dbo].[tr_AppUser]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TRIGGER [dbo].[tr_AppUser] ON [dbo].[AppUser] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info				Varchar(25),
			@SeqNo				Int,
			@Max				Int,
			@Min				Int,
			@iEmpID				Varchar(10),
			@iFirstName			Varchar(150),
			@iMiddleName		Varchar(150),
			@iLastName			Varchar(150),
			@iActive			Varchar(5),
			@iTerminated		Varchar(5),
			@iMgrID				Varchar(10),
			@iMgrName			Varchar(150),
			@iLoginID			Varchar(30),
			@iEmail				Varchar(255),
			@iCountryCode		Varchar(8),
			@iJobLevel			Varchar(8),
			@iBusinessUnit		Varchar(255),
			@iWorkPhone			Varchar(100),
			@iJobTitle			Varchar(255),
			@cEmpID				Varchar(10),
			@cFirstName			Varchar(150),
			@cMiddleName		Varchar(150),
			@cLastName			Varchar(150),
			@cActive			Varchar(5),
			@cTerminated		Varchar(5),
			@cMgrID				Varchar(10),
			@cMgrName			Varchar(150),
			@cLoginID			Varchar(30),
			@cEmail				Varchar(255),
			@cCountryCode		Varchar(8),
			@cJobLevel			Varchar(8),
			@cBusinessUnit		Varchar(255),
			@cWorkPhone			Varchar(100),
			@cJobTitle			Varchar(255)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(AU.EmpID) 
	From AppUser AU INNER JOIN deleted DEL 
	ON AU.AppUserID = DEL.AppUserID 
	AND AU.EmpId = DEL.EmpID) > 0

	Begin
	/*****************************************************************/
	/*                                                               */
	/*****************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo		Int		Identity,
			AppUserID	Int		NOT NULL,
			EmpID		Varchar(10) NOT NULL	
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	AppUserID,
				EmpId
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(AppUserID),
				@SeqNo	= 1,
				@Min	= 1
		From	#Temp

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		WHILE @SeqNo <= @Max
			 Begin		-- BEGIN WHILE LOOP
				/*****************************************************/
				/*                                                   */
				/*****************************************************/				 
				SELECT	@iEmpID			= D.EmpId,
						@iFirstName		= ISNULL(D.FirstName,''),
						@iMiddleName	= ISNULL(D.MiddleName,''),
						@iLastName		= ISNULL(D.LastName,''),
						@iActive		= ISNULL(D.Active,''),
						@iTerminated	= ISNULL(D.Terminated,''),
						@iMgrID			= ISNULL(D.MgrId,''),		
						@iMgrName		= ISNULL(D.MgrName,''),
						@iLoginID		= ISNULL(D.LoginID,''),
						@iEmail			= ISNULL(D.Email,''),
						@iCountryCode	= ISNULL(D.CountryCode,''),
						@iJobLevel		= ISNULL(D.JobLevel,''),
						@iBusinessUnit	= ISNULL(D.BuisnessUnit,''),
						@iWorkPhone		= ISNULL(D.WorkPhone,''),
						@iJobTitle		= ISNULL(D.JobTitle,'')
				From	Deleted D INNER JOIN AppUser A ON A.AppUserID = D.AppUserID 
						AND A.EmpId = D.EmpID			 
				Where	D.AppUserID IN (Select AppUserID From #Temp Where SeqNo = @SeqNo)
				And		D.EmpID IN (Select EmpID From #Temp Where SeqNo = @SeqNo)
				
				/*****************************************************/
				/*                                                   */
				/*****************************************************/				
				SELECT	@cEmpID			= A.EmpId,
						@cFirstName		= ISNULL(A.FirstName,''),
						@cMiddleName	= ISNULL(A.MiddleName,''),
						@cLastName		= ISNULL(A.LastName,''),
						@cActive		= ISNULL(A.Active,''),
						@cTerminated	= ISNULL(A.Terminated,''),
						@cMgrID			= ISNULL(A.MgrId,''),		
						@cMgrName		= ISNULL(A.MgrName,''),
						@cLoginID		= ISNULL(A.LoginID,''),
						@cEmail			= ISNULL(A.Email,''),
						@cCountryCode	= ISNULL(A.CountryCode,''),
						@cJobLevel		= ISNULL(A.JobLevel,''),
						@cBusinessUnit	= ISNULL(A.BuisnessUnit,''),
						@cWorkPhone		= ISNULL(A.WorkPhone,''),
						@cJobTitle		= ISNULL(A.JobTitle,'')
				From	Deleted D INNER JOIN AppUser A ON A.AppUserID = D.AppUserID 
						AND A.EmpId = D.EmpID	
				Where	A.AppUserID IN (Select AppUserID From #Temp Where SeqNo = @SeqNo)
				And		A.EmpID IN (Select EmpID From #Temp Where SeqNo = @SeqNo)
					
				/*****************************************************/
				/*                                                   */
				/*****************************************************/		
				IF  (
					(@cFirstName = @iFirstName) And (@cMiddleName = @iMiddleName) And (@cLastName = @iLastName) And
					(@cMgrId = @iMgrId) And (@cMgrName = @iMgrName) And (@cLoginID = @iLoginID) And (@cEmail = @iEmail) And
					(@cCountryCode = @iCountryCode) And (@cJobLevel = @iJobLevel) And (@cBusinessUnit = @iBusinessUnit) And
					(@cWorkPhone = @iWorkPhone) And (@cJobTitle = @iJobTitle)
					)
				 Begin	
					/*************************************************/
					/*					No Change(s)                 */
					/*************************************************/
					GOTO NextRecordCheck
					
				 End
				Else
				 Begin
					/*************************************************/
					/*				Update Change(s)                 */
					/*************************************************/
					Insert	AuditAppUser
					SELECT	D.AppUserID,
							D.EmpId,
							D.FirstName,
							D.MiddleName,
							D.LastName,
							D.Active,
							D.Terminated,
							D.MgrId,		
							D.MgrName,
							D.LoginID,
							D.Email,
							D.CountryCode,
							D.JobLevel,
							D.BuisnessUnit,
							D.LastUpdated,
							D.WorkPhone,
							D.JobTitle,
							@Info,
							GETDATE(),
							USER_NAME()
					From	Deleted D INNER JOIN AppUser A ON A.AppUserID = D.AppUserID 
							AND A.EmpId = D.EmpID			 
					Where	D.AppUserID IN (Select AppUserID From #Temp Where SeqNo = @SeqNo)
					And		D.EmpID IN (Select EmpID From #Temp Where SeqNo = @SeqNo)				 
				 
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
 ELSE
	Begin		 
			/*********************************************************/
			/*                                                       */
			/*********************************************************/	
			Set	@Info =  'Delete'

			/*********************************************************/
			/*                                                       */
			/*********************************************************/				
			Insert	AuditAppUser
			SELECT	AppUserID,
					EmpId,
					FirstName,
					MiddleName,
					LastName,
					Active,
					Terminated,
					MgrId,		
					MgrName,
					LoginID,
					Email,
					CountryCode,
					JobLevel,
					BuisnessUnit,
					LastUpdated,
					WorkPhone,
					JobTitle,
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


